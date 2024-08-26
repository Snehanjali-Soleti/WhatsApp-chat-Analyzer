import streamlit as st
import preprocessor, helper, sentimentAnalysis, summary  # helping files
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile
import numpy as np

# Function to save PDF
def save_pdf(content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    for line in content:
        # Encode Unicode strings as UTF-8
        line_utf8 = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, line_utf8)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name

# Function to save summary as PDF
def save_summary_as_pdf(summary_text):
    pdf_content = [
        "Summary of the Conversation\n\n",
        summary_text
    ]
    pdf_path = save_pdf(pdf_content, "Summary.pdf")
    return pdf_path

st.sidebar.title("WhatsApp Chat Analyzer")  # app left part

uploaded_file = st.sidebar.file_uploader("Choose a file")  # file uploader
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    preprocessed_data = summary.preprocess_text(data)

    if st.sidebar.button('Summary'):
        st.markdown("""
            <style>
                .title {
                    font-size: 3em;
                    color: brown;
                    text-align: center;
                    margin-bottom: 20px;
                    font-weight:bold;
                }
            </style>
        """, unsafe_allow_html=True)

        summarized_text = summary.summarize_text(preprocessed_data)
        st.markdown("<div class='title' id='summary-section'>Summary Of The Conversation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color:#f3ece9; padding:20px; border-radius:5px; font-size:17px; margin-bottom:10vh">
                {summarized_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Add button to extract summary as PDF
        pdf_path = save_summary_as_pdf(summarized_text)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download Summary PDF",
                data=pdf_file,
                file_name="Summary.pdf",
                mime="application/pdf"
            )

# The rest of your code continues here...


    if st.sidebar.button("Show Sentiment Analysis wrt to user"):
               # Read the uploaded file content
            chat_content = uploaded_file.read().decode("utf-8")

    # Load the classifier
            classifier = 'model.pkl'

    # Perform sentiment analysis
            output, pos_count, neg_count,opinion = sentimentAnalysis.perform_sentiment_analysis(chat_content)
            st.markdown("""
                <style>
                    .title {
                        font-size: 3em;
                        color: brown;
                        text-align: center;
                        margin-bottom: 20px;
                        font-weight:bold;
                    }
                    .positive {
                        color: green; 
                        font-weight: bold;
                        font-size:2em;
                    }
                    .negative {
                        color: red; 
                        font-weight: bold;
                        font-size:2em;
                    }
                    .normalText{
                        color: black;
                        margin: 20px;
                    }
                    .box{
                       display: flex;
                justify-content: space-between;
                align-items: center;  
                    }
                </style>
            """, unsafe_allow_html=True)

            
            if output:
                st.markdown("<div class='title'>Sentiment Analysis Results</div>", unsafe_allow_html=True)
                st.markdown(output, unsafe_allow_html=True)
                st.markdown(f"<div style='margin-bottom:10vh' class='box'><span class='normalText'>Total Positives : <span class='positive'>{pos_count}</span> </span> <span class='normalText' style='margin-left:330px;'>Total Negatives : <span class='negative'>{neg_count}</span> </span> </div>", unsafe_allow_html=True)
                pdf_content = [
                "Sentiment Analysis Results\n\n",
                output,
                f"Total Positives: {pos_count}",
                f"Total Negatives: {neg_count}"
                ]
                # pdf_path = save_pdf(pdf_content, "Sentiment_Analysis.pdf")
                # with open(pdf_path, "rb") as pdf_file:
                #   st.download_button(label="Download Sentiment Analysis PDF", data=pdf_file, file_name="Sentiment_Analysis.pdf", mime="application/pdf")
            else:
                st.error("Error occurred while processing the sentiment analysis.")

            neg_count = abs(neg_count)
            labels = ['positive', 'negative']
            sizes = [pos_count, neg_count]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Pie chart representation')
            st.pyplot(fig1)

            def generate_bar_graph(opinion):
                names, positive, negative = [], [], []
                for name in opinion:
                    names.append(name)
                    positive.append(opinion[name][0])
                    negative.append(opinion[name][1])
                
                def autolabel(rects, ax):
                    for rect in rects:
                        h = rect.get_height()
                        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h), ha='center', va='bottom')
                
                ind = np.arange(len(names))
                width = 0.3
                max_x = max(max(positive), max(negative)) + 2

                fig, ax = plt.subplots()
                
                rects1 = ax.bar(ind, positive, width, color='g')
                rects2 = ax.bar(ind + width, negative, width, color='r')

                ax.set_xlabel('Names')
                ax.set_ylabel('Sentiment')
                ax.set_xticks(ind + width / 2)
                ax.set_yticks(np.arange(0, max_x, 1))
                ax.set_xticklabels(names)
                ax.legend((rects1[0], rects2[0]), ('positive', 'negative'))
                plt.title('Individual User Representaion')

                autolabel(rects1, ax)
                autolabel(rects2, ax)

                st.pyplot(fig)


            generate_bar_graph(opinion)
            
            pdf_path = save_pdf(pdf_content, "Sentiment_Analysis.pdf")
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(label="Download Sentiment Analysis PDF", data=pdf_file, file_name="Sentiment_Analysis.pdf", mime="application/pdf")
          

    # messages display        
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    # user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")


    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list) #select box- dropdown - returns user

    if st.sidebar.button("Show Analysis"):    
       #
       #  <p><span style='color:brown; font-size:20px; font-weight:bold; display:inline-block; width:80px;'>{name} :</span> <span style='border: 1px solid {sentiment_color}; padding: 8px; margin: 2px; border-radius: 5px; background-color: {bsentiment_color};display:inline-block; width:600px;'>{chat}</span></p>               # button
        st.markdown(f"<div style='margin-bottom:10vh; font-size:40px; font-weight:bold; text-align: center; border: 1px solid brown;padding: 8px;border-radius: 5px; background-color: #FBE9E7; color:brown'> Chat Analysis </div>", unsafe_allow_html=True)    
        st.dataframe(df)                 #displaying dataframe
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")  #giving title
        st.markdown(f"<div style=' font-size:18px;  text-align: center; padding: 8px;border-radius: 5px; color:brown; background-color: #FBE9E7;'> Total no.of messages, words, media shared, links shared </div>", unsafe_allow_html=True)    
                            
        col1, col2, col3, col4 = st.columns(4)            #creating columns ....single row four columns

        with col1:
            st.header("Total Messages")   #title
            st.title(num_messages)       #followed analysis
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)    
        st.markdown(f"<div style=' font-size:18px;  text-align: center; padding: 8px;border-radius: 5px; color:brown; background-color: #FBE9E7;'>The rate of conversations over the time intervals such as monthly, daily</div>", unsafe_allow_html=True)    
        
        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.markdown("<div style='font-size:18px; text-align: center; padding: 8px; border-radius: 5px; color:brown; background-color: #FBE9E7;'>what is the most busiest day?<br>what is the most busiest month?<br>what is the most busiest week?<br>who is the most busiest user?</div>", unsafe_allow_html=True)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()          #uses mathplotlib

            col1, col2 = st.columns(2)              #creating 2 columns

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)         #just printing details
        st.markdown("<div style='font-size:18px; text-align: center; padding: 8px; border-radius: 5px; color:brown; background-color: #FBE9E7;'>Maximum words used</div>", unsafe_allow_html=True)

        # WordCloud
        st.title("Wordcloud")          # [3.7] pip install WordCloud
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)               
        st.pyplot(fig)              #display of wordcloud

        # most common words
        most_common_df,temp = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)
        st.markdown("<div style='font-size:18px; text-align: center; padding: 8px; border-radius: 5px; color:brown; background-color: #FBE9E7;'>Overall emoji's count</div>", unsafe_allow_html=True)
        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            # ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f")
            ax.axis('equal')
            st.pyplot(fig)
        
        pdf_content = [
            "Chat Analysis Results\n\n",
            f"Total Messages: {num_messages}",
            f"Total Words: {words}",
            f"Media Shared: {num_media_messages}",
            f"Links Shared: {num_links}",
            "\nMonthly Timeline:\n",
            timeline.to_string(index=False),
            "\nDaily Timeline:\n",
            daily_timeline.to_string(index=False),
            "\nMost Busy Day:\n",
            busy_day.to_string(),
            "\nMost Busy Month:\n",
            busy_month.to_string(),
            "\nWeekly Activity Map:\n",
            user_heatmap.to_string(),
            "\nMost Common Words:\n",
            most_common_df.to_string(index=False)
        ]
        pdf_path = save_pdf(pdf_content, "Chat_Analysis.pdf")
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="Download Chat Analysis PDF", data=pdf_file, file_name="Chat_Analysis.pdf", mime="application/pdf")


 # Function to write the review details to a text file
# Main function
st.markdown("""
        <style>
            .hero {
                background-color: #e8f5e9;
                padding: 50px;
                text-align: center;
                border-radius: 10px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                width: 60vw;
                height: 50vh;
                position: relative;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .hero h1 {
                font-size: 3em;
                color: #2e7d32;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            }
            .hero p {
                font-size: 1.2em;
                color: #388e3c;
                margin-bottom: 30px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
            }
            .review-button-container {
                position:absolute;
                bottom: 10px;
                right: 1px;
            }
            .review-button {
                color: black;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 1.2em;
                border-radius: 5px;
                cursor: pointer;
            }
            .review-button:hover {
                background-color: #388e3c;
                color:white;
            }
            .container {
                width: 50vw;
                height: 90vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>
        <div class="container">
            <div class="hero">
                <h1>WhatsApp Chat Analyzer</h1>
                <p>Analyze and visualize your WhatsApp chat data with ease. Get insights into conversation summaries, sentiment analysis, and more!</p>
            </div>
        </div>
        <div class="review-button-container">
                <a href="#review-section" class="review-button">Go to Review</a>
        </div>
    """, unsafe_allow_html=True)

# Function to write the review details to a text file
def write_review_to_file(review_text, additional_comments):
    with open("reviews.txt", "a") as file:
        file.write(f"Name: {name}\n")
        file.write(f"Review: {review}\n\n")

# Main app code
with st.form("review_form"):
    st.markdown("<p id='review-section' style='font-size: 28px; font-weight: bold; color: green;'>Review:</p>", unsafe_allow_html=True)
    name = st.text_input("enter your name:")
   # st.markdown("<p style='font-size: 18px; font-weight: bold; color: green;'>Additional Comments:</p>", unsafe_allow_html=True)
    review = st.text_area("Write your review:")
    submitted = st.form_submit_button("Submit")

    if submitted:
            # Write the review details to a text file
        write_review_to_file(name,review)
        st.markdown("<input type='text' value='Review submitted successfully!' style='font-size: 16px; font-weight: bold; color: green;' readonly>", unsafe_allow_html=True)